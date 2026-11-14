module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> (tensor<f32>, tensor<i32>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %10 = stablehlo.constant dense<[[false, true, true], [false, false, true], [false, false, false]]> : tensor<3x3xi1>
    %11 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %14 = stablehlo.constant dense<"0x0000803F0000000000000000000000000000803F0000000000000000000000000000803F"> : tensor<3x3xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %16 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<3x3xi1>) -> tensor<3x3xi1>
    %17 = stablehlo.constant dense<0.00000286102294921875> : tensor<f32>
    %18 = stablehlo.constant dense<"0x1800803F"> : tensor<f32>
    %19 = stablehlo.abs %arg1 : tensor<3x3xf32>
    %20 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %21 = stablehlo.compare LE, %19, %20 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %22 = stablehlo.constant dense<true> : tensor<i1>
    %23 = stablehlo.reduce(%21 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3x3xi1>, tensor<i1>) -> tensor<3xi1>
    %24 = stablehlo.reduce(%23 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %25 = stablehlo.iota dim = 0 : tensor<3x3xf32>
    %26 = stablehlo.iota dim = 1 : tensor<3x3xf32>
    %27 = stablehlo.compare EQ, %25, %26 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %28 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
    %29 = stablehlo.select %27, %arg1, %28 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %31 = stablehlo.reduce(%29 init: %30) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %32 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %33 = stablehlo.compare GT, %31, %32 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %34 = stablehlo.reduce(%33 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %35 = stablehlo.select %16, %arg1, %13 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %36 = stablehlo.compare EQ, %35, %13 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %37 = stablehlo.reduce(%36 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3x3xi1>, tensor<i1>) -> tensor<3xi1>
    %38 = stablehlo.reduce(%37 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %39 = stablehlo.and %24, %34 : tensor<i1>
    %40 = stablehlo.and %39, %38 : tensor<i1>
    %41 = stablehlo.select %40, %arg1, %15 : (tensor<i1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %42 = stablehlo.multiply %41, %41 : tensor<3x3xf32>
    %43 = stablehlo.reduce(%42 init: %30) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %44 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %45 = stablehlo.subtract %43, %44 : tensor<3xf32>
    %46 = stablehlo.abs %45 : tensor<3xf32>
    %47 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %48 = stablehlo.compare LE, %46, %47 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %49 = stablehlo.reduce(%48 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %50 = stablehlo.and %40, %49 : tensor<i1>
    %51 = stablehlo.select %50, %41, %15 : (tensor<i1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %52 = stablehlo.select %27, %51, %28 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %53 = stablehlo.reduce(%52 init: %30) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %54 = stablehlo.select %50, %arg0, %11 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %55 = stablehlo.constant dense<2.0> : tensor<f32>
    %56 = stablehlo.multiply %55, %54 : tensor<f32>
    %57 = stablehlo.slice %53 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
    %59 = stablehlo.log %58 : tensor<f32>
    %60 = stablehlo.constant dense<-1.0> : tensor<f32>
    %61 = stablehlo.add %56, %60 : tensor<f32>
    %62 = stablehlo.multiply %61, %59 : tensor<f32>
    %63 = stablehlo.slice %53 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %64 = stablehlo.reshape %63 : (tensor<1xf32>) -> tensor<f32>
    %65 = stablehlo.log %64 : tensor<f32>
    %66 = stablehlo.constant dense<-2.0> : tensor<f32>
    %67 = stablehlo.add %56, %66 : tensor<f32>
    %68 = stablehlo.multiply %67, %65 : tensor<f32>
    %69 = stablehlo.add %62, %68 : tensor<f32>
    %70 = stablehlo.add %56, %1 : tensor<f32>
    %71 = stablehlo.multiply %70, %55 : tensor<f32>
    %72 = stablehlo.constant dense<0.5> : tensor<f32>
    %73 = stablehlo.add %54, %72 : tensor<f32>
    %74 = stablehlo.multiply %55, %73 : tensor<f32>
    %75 = chlo.lgamma %73 : tensor<f32> -> tensor<f32>
    %76 = stablehlo.multiply %55, %75 : tensor<f32>
    %77 = chlo.lgamma %74 : tensor<f32> -> tensor<f32>
    %78 = stablehlo.subtract %76, %77 : tensor<f32>
    %79 = stablehlo.multiply %55, %78 : tensor<f32>
    %80 = stablehlo.multiply %61, %11 : tensor<f32>
    %81 = stablehlo.add %71, %80 : tensor<f32>
    %82 = stablehlo.add %54, %1 : tensor<f32>
    %83 = stablehlo.multiply %55, %82 : tensor<f32>
    %84 = chlo.lgamma %82 : tensor<f32> -> tensor<f32>
    %85 = stablehlo.multiply %55, %84 : tensor<f32>
    %86 = chlo.lgamma %83 : tensor<f32> -> tensor<f32>
    %87 = stablehlo.subtract %85, %86 : tensor<f32>
    %88 = stablehlo.multiply %11, %87 : tensor<f32>
    %89 = stablehlo.add %79, %88 : tensor<f32>
    %90 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %91 = stablehlo.multiply %81, %90 : tensor<f32>
    %92 = stablehlo.add %91, %89 : tensor<f32>
    %93 = stablehlo.negate %92 : tensor<f32>
    %94 = stablehlo.add %69, %93 : tensor<f32>
    %95 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %96 = stablehlo.select %50, %94, %95 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %96, %5 : tensor<f32>, tensor<i32>
  }
}
