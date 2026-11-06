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

local function rect(x, y, w, h, color, alpha)
    return string.format(
        "{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H%s&\\alpha&H%02X&\\p1}m %d %d l %d %d %d %d %d %d{\\p0}",
        color, alpha, x, y, x + w, y, x + w, y + h, x, y + h
    )
end

local function mouse_position()
    local mouse = mp.get_property_native("mouse-pos")
    if not mouse or not mouse.hover then
        return nil
    end
    return mouse
end

local function seek_from_mouse(mouse, width)
    local left = 70
    local right = width - 70
    if mouse.x < left or mouse.x > right then
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

    local width = mp.get_property_number("osd-width", 1280)
    local height = mp.get_property_number("osd-height", 720)
    if width <= 0 or height <= 0 then
        return
    end
    overlay.res_x = width
    overlay.res_y = height

    local mouse = mouse_position()
    local hovered = mouse ~= nil
    if not hovered and not dragging then
        if visible then
            overlay.hidden = true
            overlay:update()
            visible = false
        end
        return
    end

    local left = 70
    local right = width - 70
    local bar_y = height - 46
    local bar_h = 7
    local fraction = math.max(0, math.min(1, position / duration))
    local fill_w = math.floor((right - left) * fraction)
    local ass = rect(left, bar_y, right - left, bar_h, "FFFFFF", 105)
    ass = ass .. rect(left, bar_y, fill_w, bar_h, "FFFFFF", 0)
    ass = ass .. rect(left + fill_w - 3, bar_y - 4, 6, bar_h + 8, "FFFFFF", 0)

    local elapsed = fmt_time(position)
    local total = fmt_time(duration)
    ass = ass .. string.format(
        "{\\an1\\pos(%d,%d)\\bord0\\shad0\\fs16\\1c&HFFFFFF&} %s",
        left, height - 70, elapsed
    )
    ass = ass .. string.format(
        "{\\an3\\pos(%d,%d)\\bord0\\shad0\\fs16\\1c&HFFFFFF&}%s ",
        right, height - 70, total
    )

    if mouse and mouse.y >= height - 90 then
        local preview = seek_from_mouse(mouse, width)
        if preview then
            local px = left + (right - left) * (preview / duration)
            ass = ass .. string.format(
                "{\\an2\\pos(%.1f,%d)\\bord2\\3c&H000000&\\shad0\\fs16\\1c&HFFFFFF&}%s",
                px, height - 96, fmt_time(preview)
            )
        end
    end

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
        local width = mp.get_property_number("osd-width", 1280)
        if mouse and mouse.y >= mp.get_property_number("osd-height", 720) - 90
            and seek_from_mouse(mouse, width) then
            dragging = true
            render()
        end
    elseif event.event == "up" and dragging then
        dragging = false
        local mouse = mouse_position()
        local width = mp.get_property_number("osd-width", 1280)
        local target = mouse and seek_from_mouse(mouse, width) or nil
        if target then
            mp.commandv("script-message", "auto-translate-jump", tostring(target - position))
        end
        render()
    end
end, { complex = true })

mp.add_periodic_timer(0.08, render)
