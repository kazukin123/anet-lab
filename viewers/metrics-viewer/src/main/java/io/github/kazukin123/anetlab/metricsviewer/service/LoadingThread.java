package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class LoadingThread extends Thread {

	private static final Logger log = LoggerFactory.getLogger(LoadingThread.class);

    private final RunRepository runRepository;
    private final MetricsRepository metricsRepository;

	private volatile boolean running = true;

	@Autowired
	public LoadingThread(RunRepository runRepository, MetricsRepository metricsRepository) {
        this.runRepository = runRepository;
        this.metricsRepository = metricsRepository;
	}

	public void request(List<String> runIds, List<String> tagKeys, int maxMsec) {
		// TODO 自動生成されたメソッド・スタブ
	}
	
	@Override
	public void run() {
		// TODO 自動生成されたメソッド・スタブ
		while (running) {
			;
		}
	}

}
