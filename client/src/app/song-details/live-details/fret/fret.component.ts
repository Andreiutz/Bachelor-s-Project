import {Component, Input} from '@angular/core';

@Component({
  selector: 'app-fret',
  templateUrl: './fret.component.html',
  styleUrls: ['./fret.component.css']
})
export class FretComponent {
    @Input() stringIndex: number;
    @Input() fretIndex: number;
    @Input() display: boolean;
}
