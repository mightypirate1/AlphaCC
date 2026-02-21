use std::io::{BufWriter, Read, Write};
use std::net::TcpStream;

const REQ_MAGIC: u8 = 0x80;
const RES_MAGIC: u8 = 0x81;

const OP_GET: u8 = 0x00;
const OP_SET: u8 = 0x01;
const OP_FLUSH: u8 = 0x08;
const OP_NOOP: u8 = 0x0a;
const OP_SETQ: u8 = 0x11; // quiet SET: no response on success

const HEADER_LEN: usize = 24;
const SET_EXTRAS_LEN: u8 = 8; // flags(4) + expiration(4)

pub struct MemcachedBinaryClient {
    stream: BufWriter<TcpStream>,
}

impl MemcachedBinaryClient {
    pub fn new(host: &str, port: u16) -> Self {
        let stream = TcpStream::connect((host, port))
            .unwrap_or_else(|e| panic!("Failed to connect to memcached at {}:{}: {}", host, port, e));
        stream
            .set_nodelay(true)
            .expect("Failed to set TCP_NODELAY");
        MemcachedBinaryClient {
            stream: BufWriter::new(stream),
        }
    }

    pub fn set(&mut self, key: &[u8], value: &[u8]) {
        self.write_set_packet(OP_SET, key, value);
        self.stream.flush().unwrap();
        self.read_response_header();
    }

    /// Pipelined batch set using SETQ (quiet) + NOOP fence.
    /// SETQ doesn't respond on success; NOOP always responds.
    /// When we get the NOOP response, all prior SETQs have completed.
    pub fn set_multi(&mut self, kvs: &[(&[u8], &[u8])]) {
        if kvs.is_empty() {
            return;
        }
        for &(key, value) in kvs {
            self.write_set_packet(OP_SETQ, key, value);
        }
        // NOOP as pipeline fence
        self.write_header(OP_NOOP, 0, 0, 0);
        self.stream.flush().unwrap();

        // Read responses: SETQs only reply on error, NOOP always replies.
        // Keep reading until we see the NOOP response.
        loop {
            let (opcode, status, body_len) = self.read_response_header();
            // Drain body if present
            if body_len > 0 {
                let mut body = vec![0u8; body_len];
                self.stream.get_mut().read_exact(&mut body).unwrap();
            }
            if opcode == OP_NOOP {
                break;
            }
            if status != 0 {
                eprintln!(
                    "memcached SETQ error: opcode=0x{:02x} status=0x{:04x}",
                    opcode, status
                );
            }
        }
    }

    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        // GET request: no extras, no value
        let body_len = key.len() as u32;
        self.write_header(OP_GET, 0, key.len() as u16, body_len);
        self.stream.write_all(key).unwrap();
        self.stream.flush().unwrap();

        let (_, status, body_len) = self.read_response_header();
        if status != 0 {
            // Drain body (e.g. "Not found" error message) to keep stream aligned
            if body_len > 0 {
                let mut body = vec![0u8; body_len];
                self.stream.get_mut().read_exact(&mut body).unwrap();
            }
            if status == 0x0001 {
                return None;
            }
            panic!("memcached GET error: status=0x{:04x}", status);
        }
        if body_len == 0 {
            return None;
        }
        let mut body = vec![0u8; body_len];
        self.stream.get_mut().read_exact(&mut body).unwrap();
        // Response body = extras(4 bytes flags) + key(0 for GET) + value
        // Skip the 4-byte extras (flags)
        Some(body[4..].to_vec())
    }

    pub fn flush_all(&mut self) {
        // FLUSH request: no extras, no key, no value
        self.write_header(OP_FLUSH, 0, 0, 0);
        self.stream.flush().unwrap();
        let (_, status, body_len) = self.read_response_header();
        if body_len > 0 {
            let mut body = vec![0u8; body_len];
            self.stream.get_mut().read_exact(&mut body).unwrap();
        }
        if status != 0 {
            panic!("memcached FLUSH error: status=0x{:04x}", status);
        }
    }

    fn write_header(&mut self, opcode: u8, extras_len: u8, key_len: u16, body_len: u32) {
        let mut header = [0u8; HEADER_LEN];
        header[0] = REQ_MAGIC;
        header[1] = opcode;
        header[2..4].copy_from_slice(&key_len.to_be_bytes());
        header[4] = extras_len;
        // header[5] = 0; // data type
        // header[6..8] = 0; // vbucket id
        header[8..12].copy_from_slice(&body_len.to_be_bytes());
        // header[12..16] = 0; // opaque
        // header[16..24] = 0; // CAS
        self.stream.write_all(&header).unwrap();
    }

    fn write_set_packet(&mut self, opcode: u8, key: &[u8], value: &[u8]) {
        let body_len = SET_EXTRAS_LEN as u32 + key.len() as u32 + value.len() as u32;
        self.write_header(opcode, SET_EXTRAS_LEN, key.len() as u16, body_len);
        // Extras: flags(4) + expiration(4), both zero
        self.stream.write_all(&[0u8; 8]).unwrap();
        self.stream.write_all(key).unwrap();
        self.stream.write_all(value).unwrap();
    }

    /// Returns (opcode, status, body_length)
    fn read_response_header(&mut self) -> (u8, u16, usize) {
        let mut header = [0u8; HEADER_LEN];
        self.stream.get_mut().read_exact(&mut header).unwrap();
        assert_eq!(header[0], RES_MAGIC, "Invalid memcached response magic");
        let opcode = header[1];
        let status = u16::from_be_bytes([header[6], header[7]]);
        let body_len = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
        (opcode, status, body_len)
    }
}
