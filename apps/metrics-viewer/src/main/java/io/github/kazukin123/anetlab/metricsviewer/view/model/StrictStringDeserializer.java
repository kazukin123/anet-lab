package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.io.IOException;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;

public final class StrictStringDeserializer extends JsonDeserializer<String> {

	@Override
	public String deserialize(JsonParser parser, DeserializationContext context)
			throws IOException {
		if (!parser.hasToken(JsonToken.VALUE_STRING)) {
			return (String) context.handleUnexpectedToken(String.class, parser);
		}
		return parser.getText();
	}
}
