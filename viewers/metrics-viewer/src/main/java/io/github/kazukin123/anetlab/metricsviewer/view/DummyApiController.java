package io.github.kazukin123.anetlab.metricsviewer.view;

import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/dummy_api")
public class DummyApiController {

	@PostMapping(value = "/metrics.json", produces = MediaType.APPLICATION_JSON_VALUE)
    public Resource postIndex() {
        return new ClassPathResource("static/dummy_api/metrics.json");
    }
}