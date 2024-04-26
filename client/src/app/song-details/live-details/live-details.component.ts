import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {NotifyService} from "../../shared/notify.service";

@Component({
  selector: 'app-live-details',
  templateUrl: './live-details.component.html',
  styleUrls: ['./live-details.component.css']
})
export class LiveDetailsComponent implements OnInit, OnDestroy {
  @Input() metadata: {times: number[][], strums: number[][]};
  grif: number[][]
  stringLabels = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
  notifySubscription: any;
  time = 0;
  currentActiveStrum = 0;

  constructor(private notifyService: NotifyService) {}

  ngOnInit() {
    this.grif = this.generateGrif();
    this.notifySubscription = this.notifyService.songTimeChanged$.subscribe(time => {
      this.time = time;
      this.currentActiveStrum = 0;
      while (this.currentActiveStrum < this.metadata.strums.length-1 && this.time > this.metadata.times[this.currentActiveStrum][1]) {
        this.currentActiveStrum += 1;
      }
    })
  }

  ngOnDestroy() {
    this.notifySubscription.unsubscribe()
  }

  generateGrif() {
    const result: number[][] = [];
    for (let i = 0; i < 6; i++) {
      const column: number[] = [];
      for (let j = 0; j < 20; j++) {
        column.push(j);
      }
      result.push(column)
    }
    return result;
  }

  isFretActive(i: number, j:number) {
      const currentStrum = this.metadata.strums[this.currentActiveStrum].slice().reverse();
      return this.time > 0 && currentStrum[i] == j;
  }
}
