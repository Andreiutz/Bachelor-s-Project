import {Component, Input, OnInit} from '@angular/core';

@Component({
  selector: 'app-strum',
  templateUrl: './strum.component.html',
  styleUrls: ['./strum.component.css']
})
export class StrumComponent implements OnInit{
  @Input() data: any[]

  ngOnInit() {
    // this.data = this.data.reverse();
    // this.data.push('Em')
    // this.data = this.data.reverse();
  }

  getData(d: any): string {
    if (typeof d === 'number' && !isNaN(d)) {
      return d>= 0 ? d.toString() : '-';
    } else {
      return d.toString()
    }
  }
}
