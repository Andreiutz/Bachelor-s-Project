import {Component, ElementRef, Input, OnInit, ViewChild} from '@angular/core';
import {Song} from "../../shared/song.model";
import {RequestService} from "../../shared/request.service";

@Component({
  selector: 'app-song-item',
  templateUrl: './song-item.component.html',
  styleUrls: ['./song-item.component.css']
})
export class SongItemComponent implements OnInit {
  @Input() song: Song;
  @ViewChild('audioPlayer') audioPlayer: ElementRef;
  @ViewChild('itemContainer') itemContainer: ElementRef;
  timeUpdateListener: any;

  constructor(private requestService: RequestService) {
  }

  showAudio = false;

  ngOnInit() {
    console.log(this.song);
  }

  play() {
    this.audioPlayer.nativeElement.play();
  }

  pause() {
    this.audioPlayer.nativeElement.pause();
  }


  onItemClick() {
    if (this.showAudio) {
      this.toggleButtonClickedState(false)
      this.audioPlayer.nativeElement.removeEventListener('timeupdate', this.timeUpdateListener);
      this.showAudio = false;
    } else {
      this.toggleButtonClickedState(true)
      this.showAudio = true;
      this.requestService.fetchAudio(this.song.id).subscribe(blob => {
        const audioBlob = new Blob([blob], {type: 'audio/wav'});
        this.audioPlayer.nativeElement.src = URL.createObjectURL(audioBlob);
        this.timeUpdateListener = () => {
          console.log('Current time:', this.audioPlayer.nativeElement.currentTime)
        };
        this.audioPlayer.nativeElement.addEventListener('timeupdate', this.timeUpdateListener)
      })
    }
  }

  toggleButtonClickedState(add: boolean) {
    if (this.itemContainer && this.itemContainer.nativeElement) {
      if (add) {
        this.itemContainer.nativeElement.classList.add('button-clicked');
      } else {
        this.itemContainer.nativeElement.classList.remove('button-clicked');
      }
    }
  }
}
