import {Injectable} from "@angular/core";
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {ISong} from "./song.interface";

@Injectable({providedIn: 'root'})
export class RequestService {
    basePath = 'http://localhost:8000'
    headers = new HttpHeaders({
      "Content-Type" : "application/json",

    })
    constructor(private http: HttpClient) {
    }

    fetchSongList() {
        return this.http.get<ISong[]>(`${this.basePath}/songs`, {headers: this.headers})
    }

    fetchAudio(audioId: string) {
      return this.http.get(`${this.basePath}/file?audio_id=${audioId}`, {responseType: 'blob'})
    }

    uploadAudio(file: File) {
      const formData = new FormData();
      formData.append('file', file)
      return this.http.post<ISong>(`${this.basePath}/upload`, formData)
    }

}
