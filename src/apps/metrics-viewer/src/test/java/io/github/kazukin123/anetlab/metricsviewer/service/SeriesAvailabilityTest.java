package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class SeriesAvailabilityTest {

	@Test
	void externalNamesRemainStable() {
		assertEquals("ok", SeriesAvailability.OK.externalName());
		assertEquals("pending", SeriesAvailability.PENDING.externalName());
		assertEquals("not_found", SeriesAvailability.NOT_FOUND.externalName());
		assertEquals("empty", SeriesAvailability.EMPTY.externalName());
	}
}
