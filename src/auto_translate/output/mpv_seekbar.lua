-- Hover-only VOD seekbar for auto-translate.
-- Position and duration are supplied by Python using the YouTube source timeline.
local overlay = mp.create_osd_overlay("ass-events")
overlay.z = 20
overlay.res_x = 1280
overlay.res_y = 720

local position = 0
local duration = 0
local dragging = false
local visible = false
local last_ass = ""

local function fmt_time(value)
    value = math.max(0, math.floor(value))
    local hours = math.floor(value / 3600)
    local minutes = math.floor((value % 3600) / 60)
    local seconds = value % 60
    if hours > 0 then
        return string.format("%d:%02d:%02d", hours, minutes, seconds)
    end
    return string.format("%02d:%02d", minutes, seconds)
end


local function mouse_position()
    local mouse = mp.get_property_native("mouse-pos")
    if not mouse or not mouse.hover then
        return nil
    end
    return mouse
end

local function video_rect()
    local dims = mp.get_property_native("osd-dimensions") or {}
    local width = tonumber(dims.w) or mp.get_property_number("osd-width", 1280)
    local height = tonumber(dims.h) or mp.get_property_number("osd-height", 720)
    local left = math.max(0, tonumber(dims.ml) or 0)
    local right = width - math.max(0, tonumber(dims.mr) or 0)
    local top = math.max(0, tonumber(dims.mt) or 0)
    local bottom = height - math.max(0, tonumber(dims.mb) or 0)

    -- Fall back to the full OSD window if mpv has not reported video margins.
    if right <= left then
        left, right = 0, width
    end
    if bottom <= top then
        top, bottom = 0, height
    end
    return left, top, right, bottom, width, height
end

local function track_bounds()
    -- Match the clickable range to the rendered text track, rather than the
    -- full video width. ASS box glyphs are approximately 0.55 em wide.
    local font_size = 35
    local video_left, _, video_right = video_rect()
    local available = math.max(1, video_right - video_left)
    local bar_chars = math.max(24, math.floor(available / (font_size * 0.55)))
    local track_width = bar_chars * font_size * 0.55
    local center = (video_left + video_right) / 2
    return center - track_width / 2, center + track_width / 2, bar_chars
end

local function seek_from_mouse(mouse, left, right)
    if mouse.x < left or mouse.x > right or right <= left then
        return nil
    end
    local fraction = (mouse.x - left) / (right - left)
    return math.max(0, math.min(duration, fraction * duration))
end

local function render()
    if duration <= 0 then
        overlay.hidden = true
        overlay:update()
        visible = false
        return
    end

    local left, top, right, bottom, width, height = video_rect()
    if width <= 0 or height <= 0 or right <= left or bottom <= top then
        return
    end
    overlay.res_x = width
    overlay.res_y = height

    local mouse = mouse_position()
    -- Restrict the seekbar and its hover zone to the actual displayed video.
    local hovered = mouse ~= nil
        and mouse.x >= left and mouse.x <= right
        and mouse.y >= math.max(top, bottom - 105)
        and mouse.y <= bottom
    if not hovered and not dragging then
        if visible then
            overlay.hidden = true
            overlay:update()
            visible = false
        end
        return
    end

    local center_x = (left + right) / 2
    local _, _, bar_chars = track_bounds()
    local bar_y = bottom - 52
    local fraction = math.max(0, math.min(1, position / duration))
    local filled = math.floor(bar_chars * fraction + 0.5)
    local empty = bar_chars - filled
    local bar = "{\\1c&HFFFFFF&}" .. string.rep("━", filled)
        .. "{\\1c&H808080&}" .. string.rep("─", empty)
    local ass = string.format(
        "{\\an2\\pos(%d,%d)\\bord3\\3c&HFFFFFF&\\shad0\\fs35\\1c&HFFFFFF&}%s",
        math.floor(center_x), bar_y, bar
    )

    local elapsed = fmt_time(position)
    local total = fmt_time(duration)
    ass = ass .. string.format(
        "{\\an2\\pos(%d,%d)\\bord0\\shad0\\fs35\\1c&HFFFFFF&}%s/%s",
        math.floor(center_x), bottom - 105, elapsed, total
    )

    if ass ~= last_ass or not visible then
        overlay.data = ass
        overlay.hidden = false
        overlay:update()
        last_ass = ass
    end
    visible = true
end

mp.register_script_message("auto-translate-seekbar-update", function(pos, dur)
    position = tonumber(pos) or position
    duration = tonumber(dur) or 0
    render()
end)

mp.add_forced_key_binding("MBTN_LEFT", "auto-translate-seekbar-click", function(event)
    if event.event == "down" then
        local mouse = mouse_position()
        local left, top, right, bottom = video_rect()
        local seek_left, seek_right = track_bounds()
        if mouse and mouse.x >= seek_left and mouse.x <= seek_right
            and mouse.y >= bottom - 105 and mouse.y <= bottom
            and seek_from_mouse(mouse, seek_left, seek_right) then
            dragging = true
            render()
        end
    elseif event.event == "up" and dragging then
        dragging = false
        local mouse = mouse_position()
        local left, top, right, bottom = video_rect()
        local seek_left, seek_right = track_bounds()
        local target = mouse and mouse.x >= seek_left and mouse.x <= seek_right
            and seek_from_mouse(mouse, seek_left, seek_right) or nil
        if target then
            mp.commandv("script-message", "auto-translate-jump", tostring(target - position))
        end
        render()
    end
end, { complex = true })

mp.add_periodic_timer(0.08, render)
