import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { SongListComponent } from './song-list/song-list.component';
import { SongDetailsComponent } from './song-list/song-details/song-details.component';
import { TabDetailsComponent } from './song-list/song-details/tab-details/tab-details.component';
import { LiveDetailsComponent } from './song-list/song-details/live-details/live-details.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SongListComponent,
    SongDetailsComponent,
    TabDetailsComponent,
    LiveDetailsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
