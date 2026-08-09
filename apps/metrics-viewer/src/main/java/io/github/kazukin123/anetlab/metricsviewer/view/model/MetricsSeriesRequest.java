package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.LinkedHashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import lombok.Data;

@Data
public class MetricsSeriesRequest {
	@JsonDeserialize(using = StrictStringDeserializer.class)
	private String runId;
	@JsonDeserialize(using = StrictStringDeserializer.class)
	private String tagKey;
	private Long fromStep;
	private Long toStep;
	private Integer maxPoints;
	private final Map<String, Object> unknownFields = new LinkedHashMap<>();

	@JsonAnySetter
	public void captureUnknownField(String name, Object value) {
		unknownFields.put(name, value);
	}
}
