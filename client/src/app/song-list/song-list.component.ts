import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {Song} from "../shared/song.model";
import {RequestService} from "../shared/request.service";
import {FileChangeEvent} from "@angular/compiler-cli/src/perform_watch";
import {error} from "@angular/compiler-cli/src/transformers/util";

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
  isFetching = false;
  fileName: string = "";
  fileChosen = false;
  selectedFile: File;

  constructor(private requestService: RequestService) {
  }

  ngOnInit() {
    this.isFetching = true;
    this.requestService.fetchSongList()
      .subscribe(response => {
        this.songs = response;
        this.filteredSongs = this.songs;
        this.isFetching = false;
    }, error => {
        alert(`Error: ${error.message}`)
      })
  }

  onSearchChange() {
    const searchInput = this.inputSearch.nativeElement.value;
    this.filteredSongs = this.songs.filter((song) => {
        return song.name.indexOf(searchInput) >= 0;
    })
  }

  onConfirmButtonClick() {
    this.isFetching = true;
    if (this.selectedFile) {
      this.requestService.uploadAudio(this.selectedFile)
        .subscribe(
          song  => {
          this.songs.push(song)
            this.isFetching = false;
        },
          error => {
            alert(`Error: ${error.message}`)})
    }
  }

  onChange(event: any) {
    this.fileName = event.target.files[0].name;
    this.selectedFile = event.target.files[0];
    this.fileChosen = true;
  }

  onUploadButtonClick() {
    this.fileInput.nativeElement.click();
  }
}
