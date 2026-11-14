module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3x3xf32>) -> (tensor<f32>, tensor<i32>) {
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %11 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.constant dense<"0x0000803F0000000000000000000000000000803F0000000000000000000000000000803F"> : tensor<3x3xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<3x3xf32>) -> tensor<3x3xf32>
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
    %32 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %33 = stablehlo.subtract %31, %32 : tensor<3xf32>
    %34 = stablehlo.abs %33 : tensor<3xf32>
    %35 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %36 = stablehlo.compare LE, %34, %35 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %37 = stablehlo.reduce(%36 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %38 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %39 = stablehlo.subtract %arg1, %38 : tensor<3x3xf32>
    %40 = stablehlo.abs %39 : tensor<3x3xf32>
    %41 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %42 = stablehlo.compare LE, %40, %41 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %43 = stablehlo.reduce(%42 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3x3xi1>, tensor<i1>) -> tensor<3xi1>
    %44 = stablehlo.reduce(%43 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %45 = stablehlo.and %24, %37 : tensor<i1>
    %46 = stablehlo.and %45, %44 : tensor<i1>
    %47 = stablehlo.select %46, %arg1, %15 : (tensor<i1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %48 = stablehlo.cholesky %47, lower = true : tensor<3x3xf32>
    %49 = stablehlo.select %27, %48, %28 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %50 = stablehlo.reduce(%49 init: %30) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %51 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %52 = stablehlo.compare GT, %50, %51 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %53 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %55 = stablehlo.compare LT, %50, %54 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %56 = stablehlo.and %52, %55 : tensor<3xi1>
    %57 = stablehlo.reduce(%56 init: %22) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %58 = stablehlo.and %46, %57 : tensor<i1>
    %59 = stablehlo.select %58, %48, %15 : (tensor<i1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %60 = stablehlo.select %27, %59, %28 : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %61 = stablehlo.reduce(%60 init: %30) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3xf32>
    %62 = stablehlo.log %61 : tensor<3xf32>
    %63 = stablehlo.reduce(%62 init: %30) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %64 = stablehlo.constant dense<2.0> : tensor<f32>
    %65 = stablehlo.multiply %64, %63 : tensor<f32>
    %66 = stablehlo.select %58, %arg0, %11 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %67 = stablehlo.subtract %66, %11 : tensor<f32>
    %68 = stablehlo.multiply %67, %65 : tensor<f32>
    %69 = stablehlo.multiply %64, %66 : tensor<f32>
    %70 = stablehlo.add %69, %1 : tensor<f32>
    %71 = stablehlo.multiply %70, %64 : tensor<f32>
    %72 = stablehlo.constant dense<0.5> : tensor<f32>
    %73 = stablehlo.add %66, %72 : tensor<f32>
    %74 = stablehlo.multiply %64, %73 : tensor<f32>
    %75 = chlo.lgamma %73 : tensor<f32> -> tensor<f32>
    %76 = stablehlo.multiply %64, %75 : tensor<f32>
    %77 = chlo.lgamma %74 : tensor<f32> -> tensor<f32>
    %78 = stablehlo.subtract %76, %77 : tensor<f32>
    %79 = stablehlo.multiply %64, %78 : tensor<f32>
    %80 = stablehlo.constant dense<-1.0> : tensor<f32>
    %81 = stablehlo.add %69, %80 : tensor<f32>
    %82 = stablehlo.multiply %81, %11 : tensor<f32>
    %83 = stablehlo.add %71, %82 : tensor<f32>
    %84 = stablehlo.add %66, %1 : tensor<f32>
    %85 = stablehlo.multiply %64, %84 : tensor<f32>
    %86 = chlo.lgamma %84 : tensor<f32> -> tensor<f32>
    %87 = stablehlo.multiply %64, %86 : tensor<f32>
    %88 = chlo.lgamma %85 : tensor<f32> -> tensor<f32>
    %89 = stablehlo.subtract %87, %88 : tensor<f32>
    %90 = stablehlo.multiply %11, %89 : tensor<f32>
    %91 = stablehlo.add %79, %90 : tensor<f32>
    %92 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %93 = stablehlo.multiply %83, %92 : tensor<f32>
    %94 = stablehlo.add %93, %91 : tensor<f32>
    %95 = stablehlo.negate %94 : tensor<f32>
    %96 = stablehlo.add %68, %95 : tensor<f32>
    %97 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %98 = stablehlo.select %58, %96, %97 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %98, %5 : tensor<f32>, tensor<i32>
  }
}
