import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {Song} from "../../shared/song.model";
import {RequestService} from "../../shared/request.service";
import {error} from "@angular/compiler-cli/src/transformers/util";
import {NotifyService} from "../../shared/notify.service";
import {max} from "rxjs";

@Component({
  selector: 'app-tab-details',
  templateUrl: './tab-details.component.html',
  styleUrls: ['./tab-details.component.css']
})
export class TabDetailsComponent implements OnInit, OnDestroy{
  @Input() metadata: {times: number[][], strums: number[][]};
  notifySubscription: any;
  time = 0;
  constructor(private notifyService: NotifyService) {}

  ngOnInit() {
    this.notifySubscription = this.notifyService.songTimeChanged$.subscribe(time => {
      this.time = time;
    })
  }

  ngOnDestroy() {
    this.notifySubscription.unsubscribe()
  }


  isActive(i: number) {
    return this.metadata.times[i][0] <= this.time && this.metadata.times[i][1] >= this.time;
  }
}
