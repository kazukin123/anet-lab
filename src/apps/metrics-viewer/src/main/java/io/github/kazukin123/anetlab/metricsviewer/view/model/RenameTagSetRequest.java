package io.github.kazukin123.anetlab.metricsviewer.view.model;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/** 閉じたschemaのbody。文字列への暗黙変換を許可しない。 */
public class RenameTagSetRequest {
	@JsonProperty("name")
	@JsonDeserialize(using = StrictStringDeserializer.class)
	private String name;

	public String name() { return name; }

	@JsonProperty("newName")
	@JsonDeserialize(using = StrictStringDeserializer.class)
	private String newName;

	public String newName() { return newName; }

	private boolean unknownFields;

	@JsonAnySetter
	public void unknown(String key, Object value) { unknownFields = true; }

	public boolean hasUnknownFields() { return unknownFields; }
}
