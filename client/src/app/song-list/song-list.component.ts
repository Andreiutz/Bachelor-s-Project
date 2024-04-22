import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {Song} from "../shared/song.model";
import {RequestService} from "../shared/request.service";

@Component({
  selector: 'app-song-list',
  templateUrl: './song-list.component.html',
  styleUrls: ['./song-list.component.css']
})
export class SongListComponent implements OnInit{
  @ViewChild('inputSearch') inputSearch: ElementRef;
  @ViewChild('fileInput') fileInput: ElementRef
  songs: Song[]
  filteredSongs: Song[] = [];
  isLoading = false;
  fileName: string = "";
  fileChosen = false;
  selectedFile: File | undefined;

  constructor(private requestService: RequestService) {
  }

  ngOnInit() {
    this.isLoading = true;
    this.fetchSongs()
  }

  fetchSongs() {
    this.requestService.fetchSongList()
      .subscribe(response => {
        this.songs = response;
        this.filteredSongs = this.songs;
        this.isLoading = false;
      }, error => {
        alert(`Error: ${error.message}`)
        this.isLoading = false;
      })
  }

  onSearchChange() {
    const searchInput = this.inputSearch.nativeElement.value;
    this.filteredSongs = this.songs.filter((song) => {
        return song.name.indexOf(searchInput) >= 0;
    })
  }

  onConfirmButtonClick() {
    this.isLoading = true;
    if (this.selectedFile) {
      this.requestService.uploadAudio(this.selectedFile)
        .subscribe(
          song  => {
            this.songs.push(song)
            this.fileName = "";
            this.selectedFile = undefined;
            this.fileChosen = false;
            this.onSearchChange()
            this.isLoading = false;
          },
          error => {
            alert(`Error: ${error.message}`)
            this.isLoading = false;
          })
    }
  }

  onFileChange(event: any) {
    this.fileName = event.target.files[0].name;
    this.selectedFile = event.target.files[0];
    this.fileChosen = true;
  }

  onUploadButtonClick() {
    this.fileInput.nativeElement.click();
  }

  onDeleteEvent($event: string) {
    this.isLoading = true;
    this.requestService.deleteSong($event)
      .subscribe(song => {
        console.log(song.id)
        this.songs = this.songs.filter(s => s.id !== song.id)
        this.onSearchChange()
        this.isLoading = false;
      }, error => {
        alert(`Error: ${error.message}`)
        this.isLoading = false;
      })
  }
}
