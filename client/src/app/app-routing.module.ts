import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {SongListComponent} from "./song-list/song-list.component";
import {SongDetailsComponent} from "./song-details/song-details.component";

const routes: Routes = [
  { path: 'home', component: SongListComponent },
  { path: 'home/:id', component: SongDetailsComponent},
  { path: '**', redirectTo: '/home', pathMatch: "full"},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
