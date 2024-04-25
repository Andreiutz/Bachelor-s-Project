import {ChangeDetectorRef, Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {AudioRecordingService} from "../../shared/audio-recording.service";

@Component({
  selector: 'app-record-audio-popup',
  templateUrl: './record-audio-popup.component.html',
  styleUrls: ['./record-audio-popup.component.css']
})
export class RecordAudioPopupComponent implements OnInit {
  isRecording = false;
  audioURL: string | null = null;
  @ViewChild('audioPlayer') audioPlayer!: ElementRef<HTMLAudioElement>;

  constructor(private audioRecordingService: AudioRecordingService,
              private cd: ChangeDetectorRef) { }

  ngOnInit() {
    this.audioRecordingService.audioBlob$.subscribe(blob => {
      this.audioURL = window.URL.createObjectURL(blob);
      this.audioPlayer.nativeElement.src = this.audioURL;
      this.cd.detectChanges();
    });
  }

  startRecording() {
    this.isRecording = true;
    this.audioRecordingService.startRecording();
  }

  stopRecording() {
    this.isRecording = false;
    this.audioRecordingService.stopRecording();
  }
}
