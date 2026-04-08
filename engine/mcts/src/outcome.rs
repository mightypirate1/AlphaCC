use alpha_cc_core::WDL;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

impl Outcome {
    pub fn flip(self) -> Self {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Loss => Outcome::Win,
            Outcome::Draw => Outcome::Draw,
        }
    }

    /// Determine outcome from a WDL (argmax).
    pub fn from_wdl(wdl: &WDL) -> Self {
        if wdl.win >= wdl.loss && wdl.win >= wdl.draw {
            Outcome::Win
        } else if wdl.loss >= wdl.win && wdl.loss >= wdl.draw {
            Outcome::Loss
        } else {
            Outcome::Draw
        }
    }
}
