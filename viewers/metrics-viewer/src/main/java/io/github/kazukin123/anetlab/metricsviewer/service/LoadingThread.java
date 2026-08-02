package io.github.kazukin123.anetlab.metricsviewer.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class LoadingThread extends Thread {

	private static final int IDLE_SLEEP_MS = 10_000;
	private static final Logger log = LoggerFactory.getLogger(LoadingThread.class);

	private final IngestScheduler scheduler;
	private final int idleSleepMs;
	private volatile boolean running = true;

	public LoadingThread(IngestScheduler scheduler) {
		this(scheduler, IDLE_SLEEP_MS);
	}

	LoadingThread(IngestScheduler scheduler, int idleSleepMs) {
		this.scheduler = scheduler;
		this.idleSleepMs = idleSleepMs;
		setName("Metrics-LoadingThread");
		setDaemon(true);
	}

	public void terminate() {
		running = false;
		interrupt();
	}

	public void terminateAndWait(long timeoutMs) {
		terminate();
		if (Thread.currentThread() == this) return;
		try {
			join(timeoutMs);
			if (isAlive()) log.warn("LoadingThread did not stop within {}ms.", timeoutMs);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	@Override
	public void run() {
		log.info("LoadingThread started.");
		try {
			while (running && !isInterrupted()) {
				try {
					// 全Runが停止状態のときだけpoll間隔を空ける。
					if (!scheduler.runCycle()) Thread.sleep(idleSleepMs);
				} catch (RuntimeException e) {
					// cycle境界の予期しない失敗を記録し、HTTPを生かしたまま次cycleで回復を試みる。
					log.error("LoadingThread cycle failed; retrying in {}ms.", idleSleepMs, e);
					Thread.sleep(idleSleepMs);
				}
			}
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		} finally {
			log.info("LoadingThread stopped.");
		}
	}

}
