package com.example.server.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

@RestController
@RequestMapping("/api/files")
public class TestController {

    @PostMapping("/upload")
    public ResponseEntity<String> handleFileUpload(@RequestParam("file") MultipartFile file) throws IOException {
        System.out.println();

        if (file != null && checkWavExtension(file)) {

            String generatedFileName = UUID.randomUUID().toString();

            String rootPath = System.getProperty("user.dir").replace("server", "data") + "\\audio\\" + generatedFileName + "\\";

            Files.createDirectory(Paths.get(rootPath));

            Files.copy(file.getInputStream(), Path.of(rootPath, "audio.wav"), StandardCopyOption.REPLACE_EXISTING);

            return new ResponseEntity<>("File uploaded successfully! " + generatedFileName , HttpStatus.OK);
        } else {
            return new ResponseEntity<>("File type forbidden", HttpStatus.FORBIDDEN);
        }
    }

    private boolean checkWavExtension(MultipartFile file) {
        return file.getOriginalFilename().toLowerCase().endsWith(".wav");
    }
}
