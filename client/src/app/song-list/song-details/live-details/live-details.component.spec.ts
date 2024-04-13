import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LiveDetailsComponent } from './live-details.component';

describe('LiveDetailsComponent', () => {
  let component: LiveDetailsComponent;
  let fixture: ComponentFixture<LiveDetailsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [LiveDetailsComponent]
    });
    fixture = TestBed.createComponent(LiveDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
