package io.github.kazukin123.anetlab.metricsviewer.model;

import java.io.Serializable;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class Tag implements Comparable<Tag> , Serializable {
    private final String key;
    private final String type;

    public Tag() {
    	key = null;
    	type = null;
    }
    
    @Override
    public int compareTo(Tag o) {
        int cmp = this.key.compareTo(o.key);
        if (cmp != 0) return cmp;
        if (this.type == null && o.type == null) return 0;
        if (this.type == null) return -1;
        if (o.type == null) return 1;
        return this.type.compareTo(o.type);
    }

}
