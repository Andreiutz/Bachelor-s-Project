import {Component, OnInit} from '@angular/core';
import {RequestService} from "./shared/request.service";
import {Song} from "./shared/song.model";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  constructor(private requestService: RequestService) {
  }


  ngOnInit(): void {
  }
}
