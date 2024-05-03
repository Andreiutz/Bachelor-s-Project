import {ChangeDetectorRef, Component, ElementRef, EventEmitter, OnInit, Output, ViewChild} from '@angular/core';
import {AudioRecordingService} from "../../shared/audio-recording.service";
import {RequestService} from "../../shared/request.service";
import {DialogRef} from "@angular/cdk/dialog";
import {ISong} from "../../shared/song.interface";

@Component({
  selector: 'app-record-audio-popup',
  templateUrl: './record-audio-popup.component.html',
  styleUrls: ['./record-audio-popup.component.css']
})
export class RecordAudioPopupComponent implements OnInit {
  @Output() songAdded = new EventEmitter<ISong>();
  @ViewChild('audioPlayer') audioPlayer!: ElementRef<HTMLAudioElement>;
  @ViewChild('selectAudioInput') audioInput: ElementRef<HTMLSelectElement>;
  @ViewChild('audioNameInput') nameInput: ElementRef<HTMLInputElement>;
  blob: Blob;
  isRecording = false;
  recordStartLoading = false;
  isLoading = false;
  soundRecorded = false;
  manualStop = false;
  startTimer = 0;
  recordTimer = 0;
  recordingCountdown = 0;
  audioURL: string | null = null;
  recordingDeviceId = 'default';
  recordingDevices: {id: string, label: string}[] = []

  constructor(private audioRecordingService: AudioRecordingService,
              private requestService: RequestService,
              private cd: ChangeDetectorRef,
              private dialogRef: DialogRef<any>) { }

  ngOnInit() {
    this.audioRecordingService.audioBlob$.subscribe(blob => {
      this.blob = blob;
      this.audioURL = window.URL.createObjectURL(blob);
      this.audioPlayer.nativeElement.src = this.audioURL;
      this.cd.detectChanges();
    });
    this.audioRecordingService.getRecordingDevices().then(result => {
      this.recordingDevices = result;
      console.log(this.recordingDevices)
    });
  }

  startRecording() {
    this.soundRecorded = false;
    this.recordingCountdown = this.recordTimer;
    this.recordStartLoading = true;
    setTimeout(() => {
      this.isRecording = true;
      this.recordStartLoading = false;
      this.recordingDeviceId = this.audioInput.nativeElement.value ? this.audioInput.nativeElement.value : 'default';
      this.audioRecordingService.startRecording(this.recordingDeviceId);

      if (!this.manualStop) {
        setTimeout(() => {
          this.stopRecording();
        }, this.recordTimer * 1000);
      }

      const intervalId = setInterval(() => {
        this.recordingCountdown--;
        if (this.recordingCountdown <= 0) {
          clearInterval(intervalId);
        }
      }, 1000);
    }, this.startTimer * 1000);
  }


  stopRecording() {
      if (this.isRecording) {
        this.isRecording = false;
        this.soundRecorded = true;
        this.recordTimer = 0;
        this.audioRecordingService.stopRecording();
      }
  }

  onStartTimerChange($event: any) {
    this.startTimer = $event.target.value;
  }
  onTimerChange($event: any) {
    this.recordTimer = $event.target.value;
  }

  onUploadButtonClick() {
    const fileName = this.nameInput.nativeElement.value;
    this.uploadAudio(fileName)
  }

  uploadAudio(fileName: string) {
    this.isLoading = true;
    const file = new File([this.blob], `${fileName}.wav`, {type: 'audio/wav'});
    this.requestService.uploadAudio(file)
      .subscribe(response => {
        this.isLoading = false;
        this.songAdded.emit(response);
        this.dialogRef.close();
      }, error => {
        alert('Error: ' + error.message)
        this.isLoading = false;
      })
  }


  onRetryButtonClick() {
    this.soundRecorded = false;
    this.manualStop = false;
    this.startTimer = 0;
    this.recordTimer = 0;
  }
}
