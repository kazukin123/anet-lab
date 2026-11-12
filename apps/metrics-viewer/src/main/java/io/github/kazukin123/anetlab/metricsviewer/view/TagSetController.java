package io.github.kazukin123.anetlab.metricsviewer.view;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.service.MetricsApiException;
import io.github.kazukin123.anetlab.metricsviewer.service.TagSetStore;
import io.github.kazukin123.anetlab.metricsviewer.view.model.ApiError;
import io.github.kazukin123.anetlab.metricsviewer.view.model.DeleteTagSetRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetTagSetsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RenameTagSetRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.SaveTagSetRequest;

/** タグセットの一覧と変更を公開する。 */
@RestController
@RequestMapping("/api")
public class TagSetController {
	private final TagSetStore store;
	private final ObjectMapper requestMapper;

	public TagSetController(TagSetStore store, ObjectMapper mapper) {
		this.store = store;
		// 不正な後続JSONを拒否する設定をタグセットに限り、他APIのparse契約を保つ。
		this.requestMapper = mapper.copy().enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
	}

	@GetMapping("/tag-sets.json")
	public GetTagSetsResponse getTagSets() {
		return store.getTagSets();
	}

	@PostMapping(value = "/tag-sets/create", consumes = "application/json")
	public GetTagSetsResponse create(@RequestBody String body) throws JsonProcessingException {
		SaveTagSetRequest request = readRequest(body, SaveTagSetRequest.class);
		checkSchema(!request.hasUnknownFields() && request.name() != null && request.tags() != null);
		return store.create(request.name(), request.tags());
	}

	@PostMapping(value = "/tag-sets/update", consumes = "application/json")
	public GetTagSetsResponse update(@RequestBody String body) throws JsonProcessingException {
		SaveTagSetRequest request = readRequest(body, SaveTagSetRequest.class);
		checkSchema(!request.hasUnknownFields() && request.name() != null && request.tags() != null);
		return store.update(request.name(), request.tags());
	}

	@PostMapping(value = "/tag-sets/rename", consumes = "application/json")
	public GetTagSetsResponse rename(@RequestBody String body) throws JsonProcessingException {
		RenameTagSetRequest request = readRequest(body, RenameTagSetRequest.class);
		checkSchema(!request.hasUnknownFields() && request.name() != null && request.newName() != null);
		return store.rename(request.name(), request.newName());
	}

	@PostMapping(value = "/tag-sets/delete", consumes = "application/json")
	public GetTagSetsResponse delete(@RequestBody String body) throws JsonProcessingException {
		DeleteTagSetRequest request = readRequest(body, DeleteTagSetRequest.class);
		checkSchema(!request.hasUnknownFields() && request.name() != null);
		return store.delete(request.name());
	}

	private void checkSchema(boolean valid) {
		if (!valid) throw new IllegalArgumentException("Request body must match the JSON schema");
	}

	private <T> T readRequest(String body, Class<T> type) throws JsonProcessingException {
		T request = requestMapper.readValue(body, type);
		checkSchema(request != null);
		return request;
	}

	@ExceptionHandler({HttpMessageNotReadableException.class, JsonProcessingException.class, IllegalArgumentException.class})
	public ResponseEntity<ApiError> invalidRequest(Exception exception) {
		// parse失敗の変換をこのControllerに限り、既存APIへ影響させない。
		String message = exception instanceof HttpMessageNotReadableException || exception instanceof JsonProcessingException
				? "Request body must match the JSON schema" : exception.getMessage();
		return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ApiError("invalid_request", message));
	}

	@ExceptionHandler(MetricsApiException.class)
	public ResponseEntity<?> apiFailure(MetricsApiException exception) {
		return ResponseEntity.status(exception.getStatus()).headers(exception.getHeaders()).body(exception.getBody());
	}
}
