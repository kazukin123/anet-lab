package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.LinkedHashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

public class SwitchWorkspaceRequest {

	@JsonDeserialize(using = StrictStringDeserializer.class)
	private String name;
	private final Map<String, Object> unknownFields = new LinkedHashMap<>();

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	@JsonAnySetter
	public void setUnknownField(String key, Object value) {
		unknownFields.put(key, value);
	}

	public Map<String, Object> getUnknownFields() {
		return unknownFields;
	}
}
