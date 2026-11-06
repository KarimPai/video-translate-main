-- Hover-only VOD seekbar for auto-translate.
-- Position and duration are supplied by Python using the YouTube source timeline.
local overlay = mp.create_osd_overlay("ass-events")
overlay.z = 20

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

    if right <= left then left, right = 0, width end
    if bottom <= top then top, bottom = 0, height end
    return left, top, right, bottom, width, height
end

local function track_bounds()
    local left, _, right = video_rect()
    local width = math.max(1, right - left)
    local track_width = width * 0.70
    local center = left + width / 2
    local x1 = math.max(left, center - track_width / 2)
    local x2 = math.min(right, center + track_width / 2)
    return x1, x2
end

local function seek_from_mouse(mouse, left, right)
    if not mouse or right <= left or mouse.x < left or mouse.x > right then
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

    -- Use the complete OSD coordinate space consistently for rendering and hit testing.
    overlay.res_x = width
    overlay.res_y = height

    local mouse = mouse_position()
    local hover_top = math.max(top, bottom - 90)
    local hovered = mouse ~= nil
        and mouse.x >= left and mouse.x <= right
        and mouse.y >= hover_top and mouse.y <= bottom

    if not hovered and not dragging then
        if visible then
            overlay.hidden = true
            overlay:update()
            visible = false
        end
        return
    end

    local seek_left, seek_right = track_bounds()
    local track_width = math.max(1, math.floor(seek_right - seek_left))
    local bar_height = 10
    local bar_y = math.max(top, math.min(bottom - bar_height - 30, bottom - 62))
    local fraction = math.max(0, math.min(1, position / duration))
    local fill_width = math.max(0, math.min(track_width, math.floor(track_width * fraction)))

    -- Exactly one ASS drawing event and one origin. The grey container and
    -- white fill are two paths inside that same drawing, so they cannot drift.
    local bar_x = math.floor(seek_left)
    local bar_local_y = math.floor(bar_y)
    local ass = string.format(
        "{\\an7\\pos(%d,%d)\\bord0\\shad0\\p1\\1c&H505050&}"
            .. "m 0 0 l %d 0 l %d %d l 0 %d l 0 0 "
            .. "{\\1c&HFFFFFF&}m 0 0 l %d 0 l %d %d l 0 %d l 0 0{\\p0}",
        bar_x, bar_local_y,
        track_width, track_width, bar_height, bar_height,
        fill_width, fill_width, bar_height, bar_height
    )

    local elapsed = fmt_time(position)
    local total = fmt_time(duration)
    ass = ass .. string.format(
        "{\\an2\\pos(%d,%d)\\bord0\\shad0\\fs35\\1c&HFFFFFF&}%s/%s",
        math.floor((left + right) / 2), math.floor(bottom - 8), elapsed, total
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
        local _, _, _, bottom = video_rect()
        local seek_left, seek_right = track_bounds()
        if mouse and mouse.y >= bottom - 90 and mouse.y <= bottom
            and seek_from_mouse(mouse, seek_left, seek_right) then
            dragging = true
            render()
        end
    elseif event.event == "up" and dragging then
        dragging = false
        local mouse = mouse_position()
        local seek_left, seek_right = track_bounds()
        local target = seek_from_mouse(mouse, seek_left, seek_right)
        if target then
            mp.commandv("script-message", "auto-translate-jump", tostring(target - position))
        end
        render()
    end
end, { complex = true })

mp.add_periodic_timer(0.08, render)
