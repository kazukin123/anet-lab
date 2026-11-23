package io.github.kazukin123.anetlab.metricsviewer.service;

public class QueryCancelledException extends RuntimeException {

	public QueryCancelledException() {
		super("Metrics query was superseded");
	}
}
