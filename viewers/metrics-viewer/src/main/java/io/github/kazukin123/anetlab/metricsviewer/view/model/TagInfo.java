package io.github.kazukin123.anetlab.metricsviewer.view.model;

import com.fasterxml.jackson.annotation.JsonInclude;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@Builder
@AllArgsConstructor
@EqualsAndHashCode(of = "key")
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TagInfo implements Comparable<TagInfo> {
	private String key;
	private String type;
	private String status;
	private TagStats stats;
	private ApiError error;

	public TagInfo() {
		this.key = null;
		this.type = null;
		this.status = null;
		this.stats = null;
		this.error = null;
	}

	public TagInfo(String key) {
		this.key = key;
		this.type = null;
		this.status = null;
		this.stats = null;
		this.error = null;
	}

	@Override
	public int compareTo(TagInfo o) {
		return this.key.compareTo(o.key);
	}
}
