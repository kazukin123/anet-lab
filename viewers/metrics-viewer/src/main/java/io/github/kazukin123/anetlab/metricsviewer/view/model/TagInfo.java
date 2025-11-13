package io.github.kazukin123.anetlab.metricsviewer.view.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@Builder
@AllArgsConstructor
@EqualsAndHashCode(of = "key")
public class TagInfo implements Comparable<TagInfo> {
	private final String key;
	private final String type;

	public TagInfo() {
		this.key = null;
		this.type = null;
	}

	public TagInfo(String key) {
		this.key = key;
		this.type = null;
	}

	@Override
	public int compareTo(TagInfo o) {
		return this.key.compareTo(o.key);
	}
}
