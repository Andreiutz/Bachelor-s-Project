import {Pipe, PipeTransform} from "@angular/core";

@Pipe({
  name: 'timeFormat'
})
export class TimeFormatPipe implements PipeTransform {
  transform(value: number): string {
    const mins = Math.floor(value / 60);
    const secs = Math.floor(value % 60);

    const minsString = mins < 10 ? `0${mins}` : `${mins}`;
    const secsString = secs < 10 ? `0${secs}` : `${secs}`;

    return `${minsString}:${secsString}`;
  }
}
