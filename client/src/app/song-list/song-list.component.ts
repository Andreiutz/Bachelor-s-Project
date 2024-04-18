import {Component, OnInit} from '@angular/core';
import {Song} from "../shared/song.model";
import {RequestService} from "../shared/request.service";

@Component({
  selector: 'app-song-list',
  templateUrl: './song-list.component.html',
  styleUrls: ['./song-list.component.css']
})
export class SongListComponent implements OnInit{
  songs: Song[]
  isFetching = true;
  constructor(private requestService: RequestService) {
  }

  ngOnInit() {
    this.isFetching = true;
    this.requestService.fetchSongList().subscribe(response => {
        this.songs = response;
        this.isFetching = false;
    })
  }

}
