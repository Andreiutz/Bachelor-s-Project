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
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import {MatIconModule} from "@angular/material/icon";
import {MatTooltipModule} from "@angular/material/tooltip";
import {MatProgressSpinnerModule} from "@angular/material/progress-spinner";

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SongListComponent,
    SongDetailsComponent,
    TabDetailsComponent,
    LiveDetailsComponent,
    SongItemComponent,
    TimeFormatPipe,
  ],
    imports: [
        BrowserModule,
        AppRoutingModule,
        HttpClientModule,
        BrowserAnimationsModule,
        MatIconModule,
        MatTooltipModule,
        MatProgressSpinnerModule
    ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
