local overlay = mp.create_osd_overlay("ass-events")
overlay.z = 20
local pos, dur, dragging, visible, last = 0, 0, false, false, ""

local function rect()
    local d = mp.get_property_native("osd-dimensions") or {}
    local w, h = tonumber(d.w) or 1280, tonumber(d.h) or 720
    local l, r = math.max(0, tonumber(d.ml) or 0), w - math.max(0, tonumber(d.mr) or 0)
    local t, b = math.max(0, tonumber(d.mt) or 0), h - math.max(0, tonumber(d.mb) or 0)
    if r <= l then l, r = 0, w end
    if b <= t then t, b = 0, h end
    return l, t, r, b, w, h
end

local function bounds()
    local l, _, r = rect()
    local width = math.max(1, r - l) * 0.70
    local x = (l + r - width) / 2
    return x, x + width
end

local function mouse()
    local m = mp.get_property_native("mouse-pos")
    if m and m.hover then return m end
end

local function timecode(v)
    v = math.max(0, math.floor(v))
    local h, m, s = math.floor(v / 3600), math.floor(v % 3600 / 60), v % 60
    if h > 0 then return string.format("%d:%02d:%02d", h, m, s) end
    return string.format("%02d:%02d", m, s)
end

local function target(m, x1, x2)
    if not m or dur <= 0 or m.x < x1 or m.x > x2 then return nil end
    return math.max(0, math.min(dur, (m.x - x1) / (x2 - x1) * dur))
end

local function render()
    if dur <= 0 then overlay.hidden = true; overlay:update(); visible = false; return end
    local l, t, r, b, w, h = rect()
    if r <= l or b <= t then return end
    overlay.res_x, overlay.res_y = w, h

    local m = mouse()
    if not dragging and not (m and m.x >= l and m.x <= r and m.y >= math.max(t, b - 90) and m.y <= b) then
        if visible then overlay.hidden = true; overlay:update(); visible = false end
        return
    end

    local x1, x2 = bounds()
    x1, x2 = math.floor(x1), math.floor(x2)
    local y = math.floor(math.max(t, math.min(b - 42, b - 62)))
    local bh = math.max(4, math.min(10, math.floor(h * 0.014)))
    local bw = x2 - x1
    local fw = math.max(0, math.min(bw, math.floor(bw * pos / dur)))

    -- Flat rectangular track and clipped fill: fixed height, same origin.
    local base = string.format("{\\an7\\pos(%d,%d)\\bord0\\shad0\\1c&H505050&\\p1}m 0 0 l %d 0 l %d %d l 0 %d l 0 0{\\p0}", x1, y, bw, bw, bh, bh)
    local fill = ""
    if fw > 0 then
        fill = string.format("{\\an7\\pos(%d,%d)\\clip(%d,%d,%d,%d)\\bord0\\shad0\\1c&HFFFFFF&\\p1}m 0 0 l %d 0 l %d %d l 0 %d l 0 0{\\p0}", x1, y, x1, y, x1 + fw, y + bh, bw, bw, bh, bh)
    end
    local tc = string.format("{\\an2\\pos(%d,%d)\\bord0\\shad0\\fs35\\1c&HFFFFFF&}%s / %s", math.floor((l + r) / 2), math.floor(b - 8), timecode(pos), timecode(dur))
    local data = base .. "\\N" .. fill .. "\\N" .. tc
    if data ~= last or not visible then
        overlay.data, overlay.hidden = data, false
        overlay:update()
        last = data
    end
    visible = true
end

mp.register_script_message("auto-translate-seekbar-update", function(p, d)
    pos, dur = tonumber(p) or pos, tonumber(d) or 0
    render()
end)

mp.add_forced_key_binding("MBTN_LEFT", "auto-translate-seekbar-click", function(e)
    local l, t, r, b = rect()
    local x1, x2 = bounds()
    if e.event == "down" then
        local m = mouse()
        if m and m.y >= b - 90 and m.y <= b and target(m, x1, x2) then dragging = true; render() end
    elseif e.event == "up" and dragging then
        dragging = false
        local v = target(mouse(), x1, x2)
        if v then mp.commandv("script-message", "auto-translate-jump", tostring(v - pos)) end
        render()
    end
end, { complex = true })

mp.add_periodic_timer(0.08, render)
