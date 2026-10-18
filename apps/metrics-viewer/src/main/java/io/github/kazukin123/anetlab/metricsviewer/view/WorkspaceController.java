package io.github.kazukin123.anetlab.metricsviewer.view;

import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.service.MetricsApiException;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetWorkspacesResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.SwitchWorkspaceRequest;

/** workspace 一覧と切替APIを提供する。 */
@RestController
@RequestMapping("/api")
public class WorkspaceController {

	private final MetricsService metricsService;

	public WorkspaceController(MetricsService metricsService) {
		this.metricsService = metricsService;
	}

	/** workspace 一覧と current workspace */
	@GetMapping("/workspaces.json")
	public GetWorkspacesResponse getWorkspaces() {
		return metricsService.getWorkspaces();
	}

	/** current workspace の切替 */
	@PostMapping(value = "/workspace", consumes = "application/json")
	public ResponseEntity<?> switchWorkspace(@RequestBody SwitchWorkspaceRequest request) {
		try {
			metricsService.switchWorkspace(request);
			return ResponseEntity.noContent().build();
		} catch (MetricsApiException e) {
			return ResponseEntity.status(e.getStatus())
					.headers(e.getHeaders())
					.body(e.getBody());
		}
	}

	/** workspace切替bodyが閉じたschemaへ変換できない場合だけ応答形式を統一する。 */
	@ExceptionHandler(HttpMessageNotReadableException.class)
	public ResponseEntity<?> handleUnreadableBody(HttpMessageNotReadableException exception) {
		return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(
				"code", "invalid_request",
				"message", "Request body must match the JSON schema"));
	}
}
