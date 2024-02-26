package com.example.server.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/files")
public class TestController {

    @PostMapping("/upload")
    public ResponseEntity<String> handleFileUpload(@RequestParam("file") MultipartFile file) {
        System.out.println();

        if (checkWavExtension(file)) {
            return new ResponseEntity<>("File uploaded successfully!", HttpStatus.OK);
        } else {
            return new ResponseEntity<>("File type forbidden", HttpStatus.FORBIDDEN);
        }
    }

    private boolean checkWavExtension(MultipartFile file) {
        return file.getOriginalFilename().toLowerCase().endsWith(".wav");
    }
}
