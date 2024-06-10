import { Component, Input, OnInit } from '@angular/core';
import { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';

@Component({
  selector: 'app-bar-chart',
  standalone: true,
  imports: [NgxEchartsDirective],
  templateUrl: './bar-chart.component.html',
  styleUrl: './bar-chart.component.scss',
})
export class BarChartComponent implements OnInit {
  @Input() title: string = '';
  @Input()
  set data(data: number[]) {
    this.mergeOptions = {
      xAxis: { type: 'category' },
      series: [{ data: data }],
    };
  }
  initOpts = {
    renderer: 'svg',
    width: 350,
    height: 225,
  };
  mergeOptions: EChartsOption = {};
  options: EChartsOption = {};

  ngOnInit(): void {
    this.options = {
      title: { text: this.title },
      legend: {},
      tooltip: {},

      xAxis: { type: 'category' },
      yAxis: {},

      series: [
        {
          type: 'bar',
        },
      ],
    };
  }
}
