package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import lombok.Data;

@Data
public class GetWorkspacesResponse {
	private String current;
	private List<String> workspaces;
}
