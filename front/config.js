// 统一入口配置：导入服务与查询服务仍是两个独立进程，页面地址只在这里维护一份。
// 反代网关模式下把 IMPORT / QUERY 改成 '/api/import'、'/api/query' 即可，页面代码不用动。
(function () {
  var host = location.hostname || '127.0.0.1';

  var GW = {
    IMPORT: 'http://' + host + ':8002',
    QUERY: 'http://' + host + ':8003'
  };

  GW.page = function (which, query) {
    var base = which === 'import' ? GW.IMPORT : GW.QUERY;
    var file = which === 'import' ? 'import.html' : 'chat.html';
    return base + '/front/' + file + (query ? '?' + query : '');
  };

  // 两个服务挂载的是同一个静态目录，所以子页可以由外壳自身同源提供，
  // 页面内部再按 IMPORT / QUERY 分别调用各自的后端；同源后外壳与子页可直接互调。
  GW.local = function (which, query) {
    var file = which === 'import' ? 'import.html' : 'chat.html';
    return location.origin + '/front/' + file + (query ? '?' + query : '');
  };

  // 在统一入口（index.html）里切 Tab；被单独打开时退回新标签页
  GW.navigate = function (which, kbId) {
    var query = kbId ? 'kb_id=' + encodeURIComponent(kbId) : '';
    try {
      if (window.parent !== window && window.parent.GW && window.parent.GW.setTab) {
        window.parent.GW.setTab(which, query);
        return;
      }
    } catch (e) {
      // 外壳与页面不同源：读不到 parent.GW，退回新标签页
    }
    window.open(GW.page(which, query), '_blank', 'noopener');
  };

  window.GW = GW;
})();
