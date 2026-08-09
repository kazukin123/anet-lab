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
public class TagInfo {
	private String key;
	private String type;
	private String status;
	private TagStats stats;
	private ApiError error;

}
