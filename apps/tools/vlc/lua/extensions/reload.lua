-- VLC extension: reload video on end
function descriptor()
    return {
        title = "Auto Reload",
        version = "1.0",
        author = "custom",
        capabilities = { "input-listener" }
    }
end

function activate()
    vlc.msg.dbg("[reload] activated")
end

function deactivate()
    vlc.msg.dbg("[reload] deactivated")
end

function meta_changed()
    vlc.msg.dbg("[reload] meta_changed")
end

-- VLC が "input-listener" に必ず要求するコールバック
function input_changed()
    vlc.msg.dbg("[reload] input_changed")
end

-- 再生状態の変化を受け取るコールバック
function input_events(event)
    vlc.msg.dbg("[reload] event: " .. tostring(event))

    if event == "end-reached" then
        vlc.msg.dbg("[reload] end reached – reloading")

        local item = vlc.input.item()
        if not item then
            vlc.msg.err("[reload] item is nil")
            return
        end

        local uri = item:uri()
        vlc.msg.dbg("[reload] uri = " .. tostring(uri))

        -- 再読み込み
        vlc.playlist.clear()
        vlc.playlist.add({{ path = uri }})
        vlc.playlist.play()
    end
end
