export class Song {
  public id: string;
  public name: string;
  public last_edited: Date;
  public duration: number;

  constructor(id: string, name: string, last_edited: Date, duration: number) {
    this.id = id;
    this.name = name;
    this.last_edited = last_edited;
    this.duration = duration
  }
}
