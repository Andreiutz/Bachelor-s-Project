import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { SongListComponent } from './song-list/song-list.component';
import {HttpClientModule} from "@angular/common/http";
import { SongItemComponent } from './song-list/song-item/song-item.component';
import {TabDetailsComponent} from "./song-details/tab-details/tab-details.component";
import {SongDetailsComponent} from "./song-details/song-details.component";
import {LiveDetailsComponent} from "./song-details/live-details/live-details.component";
import {TimeFormatPipe} from "./song-list/song-item/time.format";

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SongListComponent,
    SongDetailsComponent,
    TabDetailsComponent,
    LiveDetailsComponent,
    SongItemComponent,
    TimeFormatPipe
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
