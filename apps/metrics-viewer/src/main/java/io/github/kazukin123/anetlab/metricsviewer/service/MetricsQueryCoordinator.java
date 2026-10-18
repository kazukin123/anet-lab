package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;

@Component
public class MetricsQueryCoordinator {

	private static final long ACQUIRE_TIMEOUT_SECONDS = 5L;
	private static final int MAX_RETAINED_CHANNELS = 64;

	private final int maxConcurrentQueries;
	private final Semaphore permits;
	private final Map<QueryChannel, ChannelState> channels =
			new LinkedHashMap<>(16, 0.75f, true);
	private final Set<Ticket> liveTickets = new HashSet<>();
	private long cancelledEpochWatermark = Long.MIN_VALUE;
	private boolean terminal;

	@Autowired
	public MetricsQueryCoordinator(MetricsViewerSettings settings) {
		this(settings.getMaxConcurrentQueries());
	}

	MetricsQueryCoordinator(int maxConcurrentQueries) {
		this.maxConcurrentQueries = maxConcurrentQueries;
		this.permits = new Semaphore(maxConcurrentQueries, true);
	}

	<T> T run(QueryChannel channel, long sequence, QueryWork<T> work) {
		final Ticket ticket;
		final Ticket previous;
		synchronized (this) {
			if (terminal) throw new QueryCancelledException();
			// channel 内の新旧を確定し、新しい ticket だけを current として公開する。
			final ChannelState state = channels.computeIfAbsent(channel, ignored -> new ChannelState());
			if (state.initialized && sequence <= state.latestSequence) {
				throw new QueryCancelledException();
			}
			previous = state.current;
			ticket = new Ticket();
			liveTickets.add(ticket);
			state.initialized = true;
			state.latestSequence = sequence;
			state.current = ticket;
			if (channels.size() > MAX_RETAINED_CHANNELS) {
				final var iterator = channels.entrySet().iterator();
				iterator.next();
				iterator.remove();
			}
		}
		if (previous != null) previous.cancel();

		boolean acquired = false;
		try {
			// process-global の fair permit を取得してから query 本体を実行する。
			ticket.beginPermitWait();
			try {
				acquired = permits.tryAcquire(ACQUIRE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
			} finally {
				ticket.endPermitWait();
			}
			if (!acquired) throw new QueryCapacityException(
					"timeout", activeCount(), permits.getQueueLength());
			ticket.checkpoint();
			final T result = work.run(new QueryExecution(this, ticket));
			ticket.checkpoint();
			return result;
		} catch (InterruptedException e) {
			if (ticket.cancelled()) throw new QueryCancelledException();
			Thread.currentThread().interrupt();
			throw new QueryCapacityException(
					"interrupted", activeCount(), permits.getQueueLength());
		} finally {
			if (acquired) permits.release();
			synchronized (this) {
				// 遅れて完了した旧 ticket が新しい current を消さないよう identity を確認する。
				liveTickets.remove(ticket);
				final ChannelState state = channels.get(channel);
				if (state != null && state.current == ticket) state.current = null;
			}
		}
	}

	void cancelWorkspace(long epoch) {
		final List<Ticket> cancelling = new ArrayList<>();
		synchronized (this) {
			// 未束縛 ticket の race を閉じるため、過去最大の cancel 済み epoch を保持する。
			cancelledEpochWatermark = Math.max(cancelledEpochWatermark, epoch);
			for (Ticket ticket : liveTickets) {
				if (ticket.isBoundToEpochAtMost(cancelledEpochWatermark)) cancelling.add(ticket);
			}
		}
		for (Ticket ticket : cancelling) ticket.cancel();
	}

	void cancelAll() {
		final List<Ticket> cancelling;
		synchronized (this) {
			terminal = true;
			cancelling = new ArrayList<>(liveTickets);
		}
		for (Ticket ticket : cancelling) ticket.cancel();
	}

	void bindWorkspace(Ticket ticket, long epoch) {
		final boolean cancelled;
		synchronized (this) {
			// lease 取得直後の epoch を ticket へ束縛し、既通過の cancel と照合する。
			ticket.bindWorkspace(epoch);
			cancelled = epoch <= cancelledEpochWatermark;
		}
		if (cancelled) ticket.cancel();
		ticket.checkpoint();
	}

	private int activeCount() {
		return maxConcurrentQueries - permits.availablePermits();
	}

	private static final class ChannelState {
		private boolean initialized;
		private long latestSequence;
		private Ticket current;
	}

	static final class Ticket {
		private volatile boolean cancelled;
		private Long workspaceEpoch;
		private Thread permitWaiter;
		private boolean cancelInterruptedWaiter;
		private final Deque<Statement> activeStatements = new ArrayDeque<>();

		private void cancel() {
			final Thread waiter;
			final Statement statement;
			synchronized (this) {
				cancelled = true;
				waiter = permitWaiter;
				statement = activeStatements.peekLast();
				if (waiter != null) {
					// endPermitWait() と全順序化し、送出後にだけmarkerをclearさせる。
					cancelInterruptedWaiter = true;
					waiter.interrupt();
				}
			}
			if (statement != null) {
				try {
					statement.cancel();
				} catch (SQLException ignored) {
					// Cancellation is best-effort; checkpoints still stop Java-side work.
				}
			}
		}

		private synchronized void beginPermitWait() {
			checkpoint();
			permitWaiter = Thread.currentThread();
			if (cancelled) {
				permitWaiter = null;
				throw new QueryCancelledException();
			}
		}

		private synchronized void endPermitWait() {
			permitWaiter = null;
			if (cancelInterruptedWaiter) {
				cancelInterruptedWaiter = false;
				Thread.interrupted();
			}
		}

		private boolean cancelled() {
			return cancelled;
		}

		private synchronized void bindWorkspace(long epoch) {
			if (workspaceEpoch != null) {
				throw new IllegalStateException("Workspace epoch is already bound");
			}
			workspaceEpoch = epoch;
		}

		private synchronized boolean isBoundToEpochAtMost(long epoch) {
			return workspaceEpoch != null && workspaceEpoch <= epoch;
		}

		synchronized StatementRegistration registerStatement(Statement statement) {
			checkpoint();
			activeStatements.addLast(statement);
			return new StatementRegistration(this, statement);
		}

		synchronized void unregisterStatement(Statement statement) {
			if (activeStatements.peekLast() != statement) {
				throw new IllegalStateException("Query statements must be closed in LIFO order");
			}
			activeStatements.removeLast();
		}

		void checkpoint() {
			if (cancelled) throw new QueryCancelledException();
		}
	}
}

record QueryChannel(String value) {
}

@FunctionalInterface
interface QueryWork<T> {
	T run(QueryExecution query);
}

final class QueryExecution {

	private final MetricsQueryCoordinator coordinator;
	private final MetricsQueryCoordinator.Ticket ticket;

	QueryExecution(MetricsQueryCoordinator coordinator, MetricsQueryCoordinator.Ticket ticket) {
		this.coordinator = coordinator;
		this.ticket = ticket;
	}

	void bindWorkspace(long epoch) {
		coordinator.bindWorkspace(ticket, epoch);
	}

	void checkpoint() {
		ticket.checkpoint();
	}

	StatementRegistration registerStatement(Statement statement) {
		return ticket.registerStatement(statement);
	}
}

final class StatementRegistration implements AutoCloseable {

	private final MetricsQueryCoordinator.Ticket ticket;
	private final Statement statement;
	private boolean closed;

	StatementRegistration(MetricsQueryCoordinator.Ticket ticket, Statement statement) {
		this.ticket = ticket;
		this.statement = statement;
	}

	@Override
	public void close() {
		if (closed) return;
		ticket.unregisterStatement(statement);
		closed = true;
	}
}

final class QueryCapacityException extends RuntimeException {

	private final String reason;
	private final int active;
	private final int queued;

	QueryCapacityException(String reason, int active, int queued) {
		super("Metrics query capacity is busy");
		this.reason = reason;
		this.active = active;
		this.queued = queued;
	}

	String reason() {
		return reason;
	}

	int active() {
		return active;
	}

	int queued() {
		return queued;
	}
}
