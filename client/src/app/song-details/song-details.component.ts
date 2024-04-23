import {Component, ElementRef, Input, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {RequestService} from "../shared/request.service";
import {ISong} from "../shared/song.interface";
import {Song} from "../shared/song.model";
import {ActivatedRoute} from "@angular/router";
import {error} from "@angular/compiler-cli/src/transformers/util";
import {concatMap} from "rxjs";
import {NotifyService} from "../shared/notify.service";

@Component({
  selector: 'app-song-details',
  templateUrl: './song-details.component.html',
  styleUrls: ['./song-details.component.css']
})
export class SongDetailsComponent implements OnInit, OnDestroy{
  @ViewChild('tabButton') tabButton: ElementRef
  @ViewChild('liveButton') liveButton: ElementRef
  @ViewChild('audioPlayer') audioPlayer: ElementRef;

  song: Song = new Song('', '', new Date(), 0);
  notFound = true;
  tablatureTabSelected = true;
  isLoading = false;
  timeUpdateListener: any;

  tablatureMetadata: {strums:[][], times:[][]}
  liveMetadata: {strums:[][], times:[][]}

  constructor(private requestService: RequestService,
              private notifyService: NotifyService,
              private route: ActivatedRoute) {
  }

  ngOnInit() {
    this.initSongDetails()
    this.tablatureTabSelected = true
  }

  ngOnDestroy() {
    this.audioPlayer.nativeElement.removeEventListener('timeupdate', this.timeUpdateListener);
  }

  initSongDetails() {
    //todo also add pipe for live metadata
    this.isLoading = true;
    this.requestService.fetchSong(this.route.snapshot.params['id'])
      .pipe(
        concatMap(response => {
          //Initialize song and send strums further
          this.song = new Song(response.id, response.name, response.last_edited, response.duration)
          this.notFound = false;
          return this.requestService.fetchTabStrums(this.song.id)
        }),
        concatMap(response => {
          //Initialize strums and send the audio further
          this.tablatureMetadata = response;
          return this.requestService.fetchFullStrums(this.song.id);
        }),
        concatMap(response => {
          this.liveMetadata = response;
          return this.requestService.fetchAudio(this.song.id)
        })
      )
      .subscribe(blob => {
        //Initialize the audio
          const audioBlob = new Blob([blob], {type: 'audio/wav'});
          this.audioPlayer.nativeElement.src = URL.createObjectURL(audioBlob);
          this.timeUpdateListener = () => {
            const currentTime = this.audioPlayer.nativeElement.currentTime;
            // console.log('Current time:', currentTime)
            this.notifyService.notifyTimeChanged(currentTime)
          };
        this.audioPlayer.nativeElement.addEventListener('timeupdate', this.timeUpdateListener)
          this.isLoading = false;
      }, error => {
        this.notFound = true;
        this.isLoading = false;
      })
  }


  initLiveMetadata() {

  }

  onTabButtonClick() {
      if (!this.tablatureTabSelected) {
        this.tablatureTabSelected = true;
      }
  }

  onLiveButtonClick() {
    if (this.tablatureTabSelected) {
      this.tablatureTabSelected = false;
    }
  }
}
