package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

public class PrioritizeRunsRequest {

	@JsonDeserialize(contentUsing = StrictStringDeserializer.class)
	private List<String> runIds;
	private final Map<String, Object> unknownFields = new LinkedHashMap<>();

	public List<String> getRunIds() {
		return runIds;
	}

	public void setRunIds(List<String> runIds) {
		this.runIds = runIds;
	}

	@JsonAnySetter
	public void setUnknownField(String key, Object value) {
		unknownFields.put(key, value);
	}

	public Map<String, Object> getUnknownFields() {
		return unknownFields;
	}
}
