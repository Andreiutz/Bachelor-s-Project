import {Component, OnInit} from '@angular/core';
import {RequestService} from "../shared/request.service";
import {ISong} from "../shared/song.interface";
import {Song} from "../shared/song.model";
import {ActivatedRoute} from "@angular/router";
import {error} from "@angular/compiler-cli/src/transformers/util";

@Component({
  selector: 'app-song-details',
  templateUrl: './song-details.component.html',
  styleUrls: ['./song-details.component.css']
})
export class SongDetailsComponent implements OnInit{
  song: Song;
  notFound = true;

  constructor(private requestService: RequestService,
              private route: ActivatedRoute) {
  }

  ngOnInit() {
    this.requestService.fetchSong(this.route.snapshot.params['id'])
      .subscribe(response => {
        this.song = new Song(response.id, response.name, response.last_edited, response.duration)
        this.notFound = false;
      }, error => {
        this.notFound = true;
      })
  }



}
