package io.github.kazukin123.anetlab.metricsviewer.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class LoadingThread extends Thread {

	private static final int IDLE_SLEEP_MS = 10_000;
	private static final int PRIORITY_POLL_MS = 1_000;
	private static final Logger log = LoggerFactory.getLogger(LoadingThread.class);

	private final WorkspaceManager workspaceManager;
	private final int idleSleepMs;
	private volatile boolean running = true;
	private final Object wakeMonitor = new Object();
	private long wakeSequence;

	@Autowired
	public LoadingThread(WorkspaceManager workspaceManager) {
		this(workspaceManager, IDLE_SLEEP_MS);
	}

	LoadingThread(WorkspaceManager workspaceManager, int idleSleepMs) {
		this.workspaceManager = workspaceManager;
		this.idleSleepMs = idleSleepMs;
		workspaceManager.setIngestWakeup(this::wakeUp);
		setName("Metrics-LoadingThread");
		setDaemon(true);
	}

	public void terminate() {
		running = false;
		workspaceManager.requestIngestStop();
		wakeUp();
	}

	private void wakeUp() {
		synchronized (wakeMonitor) {
			wakeSequence++;
			wakeMonitor.notifyAll();
		}
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
				final long before;
				synchronized (wakeMonitor) {
					before = wakeSequence;
				}
				final boolean immediate = workspaceManager.runIngestCycle();
				synchronized (wakeMonitor) {
					// cycle中のwakeを失わず、SQL処理へのinterruptを使わずに待機解除する。
					if (!immediate && running && before == wakeSequence)
						wakeMonitor.wait(workspaceManager.priorityRunPaths().isEmpty()
								? idleSleepMs
								: Math.min(idleSleepMs, PRIORITY_POLL_MS));
				}
			}
		}
		catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		catch (RuntimeException e) {
			log.error("LoadingThread failed; writer stopped.", e);
		} finally {
			running = false;
			workspaceManager.requestIngestStop();
			workspaceManager.closeIngestResources();
			log.info("LoadingThread stopped.");
		}
	}
}
