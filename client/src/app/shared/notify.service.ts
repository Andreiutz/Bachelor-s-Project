import {Injectable} from "@angular/core";
import {Subject} from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class NotifyService {
    private songTimeChangedSource = new Subject<number>();
    songTimeChanged$ = this.songTimeChangedSource.asObservable();

    notifyTimeChanged(time: number) {
      this.songTimeChangedSource.next(time);
    }
}
