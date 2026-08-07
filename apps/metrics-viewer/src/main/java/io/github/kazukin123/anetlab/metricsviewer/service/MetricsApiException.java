package io.github.kazukin123.anetlab.metricsviewer.service;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;

public class MetricsApiException extends RuntimeException {

	private final HttpStatus status;
	private final Object body;
	private final HttpHeaders headers;

	public MetricsApiException(HttpStatus status, Object body) {
		this(status, body, new HttpHeaders());
	}

	public MetricsApiException(HttpStatus status, Object body, HttpHeaders headers) {
		super(body == null ? status.getReasonPhrase() : body.toString());
		this.status = status;
		this.body = body;
		this.headers = headers;
	}

	public HttpStatus getStatus() {
		return status;
	}

	public Object getBody() {
		return body;
	}

	public HttpHeaders getHeaders() {
		return headers;
	}
}
