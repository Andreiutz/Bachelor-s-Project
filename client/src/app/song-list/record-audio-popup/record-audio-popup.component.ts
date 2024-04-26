import {ChangeDetectorRef, Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {AudioRecordingService} from "../../shared/audio-recording.service";
import {RequestService} from "../../shared/request.service";

@Component({
  selector: 'app-record-audio-popup',
  templateUrl: './record-audio-popup.component.html',
  styleUrls: ['./record-audio-popup.component.css']
})
export class RecordAudioPopupComponent implements OnInit {
  @ViewChild('audioPlayer') audioPlayer!: ElementRef<HTMLAudioElement>;
  @ViewChild('selectAudioInput') audioInput: ElementRef<HTMLSelectElement>;
  @ViewChild('audioNameInput') nameInput: ElementRef<HTMLInputElement>;
  blob: Blob;
  isRecording = false;
  soundRecorded = false;
  manualStop = false;
  startTimer = 0;
  recordTimer = 0;
  audioURL: string | null = null;
  recordingDeviceId = 'default';
  recordingDevices: {id: string, label: string}[] = []

  constructor(private audioRecordingService: AudioRecordingService,
              private requestService: RequestService,
              private cd: ChangeDetectorRef) { }

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
    setTimeout(() => {
        this.isRecording = true;
        this.recordingDeviceId = this.audioInput.nativeElement.value ? this.audioInput.nativeElement.value : 'default'
        this.audioRecordingService.startRecording(this.recordingDeviceId);
        console.log(this.recordTimer)
        if (!this.manualStop) {
          setTimeout(() => {
            this.stopRecording()
          }, this.recordTimer * 1000)
        }
      }, this.startTimer * 1000)
  }

  stopRecording() {
    this.isRecording = false;
    this.soundRecorded = true;
    this.audioRecordingService.stopRecording();
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
    const file = new File([this.blob], `${fileName}.wav`, {type: 'audio/wav'});
    this.requestService.uploadAudio(file)
      .subscribe(response => {
          console.log(response)
      }, error => {
        alert('Error: ' + error.message)
      })
  }


}
